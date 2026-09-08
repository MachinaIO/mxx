//! BFV parameters, polynomial blocks, relation carriers, and scheme capabilities.
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
use mxx_dsl::{
    Bytes, DslContext, DslError, Family, FamilyType, GraphValue, GraphValueSchema, HashTag, Int,
    Mat, MatType, Ring,
};
use mxx_ir_core::{
    IntExpr, ParamEnv, ValueHandle,
    artifact::{ArtifactConfidentiality, ProductionId},
    types::{MatrixType, WireType},
};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use noise::BfvNoiseValue;
use num_bigint::{BigInt, BigUint};
use num_traits::One;
pub use public_key::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ops::{Add, Mul, Neg, Sub},
    sync::Arc,
};
use thiserror::Error;
impl From<BfvError> for crate::noise_refresh::setup::RefreshSetupError {
    fn from(error: BfvError) -> Self {
        Self::Backend(Box::new(error))
    }
}

#[derive(Debug, Error)]
pub enum BfvError {
    #[error("invalid BFV parameters: {0}")]
    Parameters(&'static str),
    #[error("BFV budget exceeded: {0}")]
    Budget(&'static str),
    #[error("BFV key, level, or matrix context mismatch")]
    Context,
    #[error("invalid BFV LUT or RHS")]
    Lut,
    #[error("BFV graph construction failed: {0}")]
    Graph(String),
}

impl From<mxx_dsl::DslError> for BfvError {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

/// Fixed low-base ciphertext satisfying r*C = p_next*nu*r + F.
/// Sampling has L=(); preprocessing supplies a required high-base lift with L=C.
/// Its plaintext is owned by preprocessing, never supplied to online evaluation.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPrivateRhs<C = BfvMatrix, L = ()> {
    pub(crate) lifted_ciphertext: L,
    pub(crate) rhs_id: [u8; 32],
    pub(crate) ciphertext: C,
    pub(crate) level: usize,
    pub(crate) context: [u8; 32],
    pub(crate) id: [u8; 32],
}

/// Cached direct product. Both ordinary states retain their key/level identity;
/// the fixed RHS lift is shared with public preparation, without LUT helpers.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPreparedPrivateMul<K = BfvPublicKey, C = BfvMatrix> {
    pub(crate) input: K,
    pub(crate) output: K,
    pub(crate) rhs: BfvPrivateRhs<C, C>,
}

impl<C, L> BfvPrivateRhs<C, L> {
    pub fn rhs_id(&self) -> &[u8; 32] {
        &self.rhs_id
    }
    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }
}

impl<K, C> BfvPreparedPrivateMul<K, C> {
    pub fn output_key(&self) -> &K {
        &self.output
    }

    pub fn rhs(&self) -> &BfvPrivateRhs<C, C> {
        &self.rhs
    }
}

impl BfvPublicKeyFamily {
    fn from_program(values: &BfvPublicProgramValue, count: IntExpr) -> Result<Self, BfvError> {
        match values {
            ProgramValue::Scalar((value, _)) => Ok(Self {
                matrices: ProgramValue::Scalar(value.matrix.clone()),
                level: value.level,
                id: value.id,
                context: value.context,
                range: FamilyRange::full(count).map_err(|_| BfvError::Context)?,
            }),
            ProgramValue::Family(values) => {
                let metadata = values.schema().element.0;
                Ok(Self {
                    matrices: ProgramValue::Family(values.field(|(value, _)| value.matrix)?),
                    level: metadata.level,
                    id: metadata.id,
                    context: metadata.context,
                    range: FamilyRange::full(values.count().clone())
                        .map_err(|_| BfvError::Context)?,
                })
            }
        }
    }
    fn into_program(self) -> Result<BfvPublicProgramValue, BfvError> {
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
                BfvPublicKey {
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
impl BfvEncodingFamily {
    fn from_program(values: &BfvEncodingProgramValue, count: IntExpr) -> Result<Self, BfvError> {
        match values {
            ProgramValue::Scalar((value, _)) => Ok(Self {
                rows: ProgramValue::Scalar(value.row.clone()),
                level: value.level,
                key_id: value.key_id,
                context: value.context,
                range: FamilyRange::full(count).map_err(|_| BfvError::Context)?,
            }),
            ProgramValue::Family(values) => {
                let metadata = values.schema().element.0;
                Ok(Self {
                    rows: ProgramValue::Family(values.field(|(value, _)| value.row)?),
                    level: metadata.level,
                    key_id: metadata.key_id,
                    context: metadata.context,
                    range: FamilyRange::full(values.count().clone())
                        .map_err(|_| BfvError::Context)?,
                })
            }
        }
    }
    fn into_program(self) -> Result<BfvEncodingProgramValue, BfvError> {
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
                BfvEncoding {
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
impl ExponentLUTFixedRhsScheme for BfvScheme {
    type FixedRhs = BfvPrivateRhs<Family<BfvMatrix>>;
    type PreparedFixedProduct =
        BfvPreparedPrivateMul<BfvPublicKeyFamily, ProgramValue<BfvMatrix, Family<BfvMatrix>>>;
    type FixedProductBinding = BfvArtifactBinding;

    fn fixed_product_binding(&self) -> Self::FixedProductBinding {
        self.artifact_binding()
    }

    fn combine_fixed_rhs(
        &self,
        rhs: &Self::FixedRhs,
        weights: &Family<Mat>,
        active: &FamilyRange,
        id: [u8; 32],
    ) -> Result<Self::FixedRhs, BfvError> {
        let ciphertext =
            crate::utils::weighted_family_combinations(&rhs.ciphertext, weights, active)?;
        Ok(BfvPrivateRhs {
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
        input: &BfvPublicProgramValue,
        rhs: &Self::FixedRhs,
        id: [u8; 32],
    ) -> Result<(BfvPublicProgramValue, Self::PreparedFixedProduct), BfvError> {
        let key = BfvPublicKeyFamily::from_program(input, rhs.ciphertext.count().clone())?;
        if key.range.start() != &0.into() || key.level != rhs.level || key.context != rhs.context {
            return Err(BfvError::Context);
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
        input: &BfvEncodingProgramValue,
        prepared: &Self::PreparedFixedProduct,
    ) -> Result<BfvEncodingProgramValue, BfvError> {
        let encoding =
            BfvEncodingFamily::from_program(input, prepared.input.range.count().clone())?;
        self.evaluate_mul_private_family(&encoding, prepared)?.into_program()
    }
}

#[derive(Clone, Debug)]
pub struct BfvParameters {
    levels: Vec<DCRTPolyParams>,
    drops: Vec<BigUint>,
    eta: usize,
}

/// Public workload inputs, independent of secret values and sampled noise.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BfvPlanRequest {
    pub ring_dimension: u32,
    pub widths: Vec<usize>,
    pub slots: Vec<usize>,
    pub eta: usize,
    pub crt_bits: usize,
    pub terminal_depth: usize,
    pub max_crt_depth: usize,
    pub memory_budget_bytes: u64,
    pub terminal_value_bound: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BfvNoiseBound {
    /// Bound on e in c = s*a - p*mu*r + e.
    pub error: BigUint,
    pub payload_l1: BigUint,
}

impl BfvNoiseBound {
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
pub struct BfvParameterManifest {
    pub workload: BfvPlanRequest,
    pub ring_dimension: u32,
    pub eta: usize,
    pub levels: Vec<Vec<u64>>,
    pub drops: Vec<BigUint>,
    pub payload_scales: Vec<BigUint>,
    pub noise_bounds: Vec<BfvNoiseBound>,
    pub estimated_bytes: BigUint,
    /// Arithmetic acceptance alone never certifies related-secret security.
    pub security_certified: bool,
}

impl BfvParameters {
    pub fn new(levels: Vec<DCRTPolyParams>, eta: usize) -> Result<Self, BfvError> {
        let first = levels.first().ok_or(BfvError::Parameters("empty modulus chain"))?;
        if levels.len() < 2 {
            return Err(BfvError::Parameters("need a modulus transition"));
        }
        for level in &levels {
            if level.ring_dimension() != first.ring_dimension() || level.dropped_moduli() != 0 {
                return Err(BfvError::Parameters("incompatible ring or partial basis"));
            }
        }
        let mut drops = Vec::with_capacity(levels.len());
        for pair in levels.windows(2) {
            let source = pair[0].to_crt().0;
            let target = pair[1].to_crt().0;
            if target.len() >= source.len() || target.iter().any(|p| !source.contains(p)) {
                return Err(BfvError::Parameters("levels must be strictly nested prime bases"));
            }
            drops.push(pair[0].modulus().as_ref() / pair[1].modulus().as_ref());
        }
        drops.push(BigUint::one());
        Ok(Self { levels, drops, eta })
    }

    pub fn levels(&self) -> &[DCRTPolyParams] {
        &self.levels
    }

    /// Register these single-prime bases alongside the evaluation chain when
    /// the runtime will execute CRT refresh at this output level.
    pub fn refresh_residue_parameters(
        &self,
        level: usize,
    ) -> Result<Vec<DCRTPolyParams>, BfvError> {
        let parameters = self.level(level)?;
        parameters
            .to_crt()
            .0
            .into_iter()
            .map(|prime| {
                parameters
                    .select_modulus(&BigUint::from(prime))
                    .ok_or(BfvError::Parameters("refresh residue is outside the CRT basis"))
            })
            .collect()
    }
    pub fn level(&self, level: usize) -> Result<&DCRTPolyParams, BfvError> {
        self.levels.get(level).ok_or(BfvError::Parameters("level outside modulus chain"))
    }
    pub fn scale(&self, level: usize) -> Result<&BigUint, BfvError> {
        self.drops.get(level).ok_or(BfvError::Parameters("level outside modulus chain"))
    }
    pub fn eta(&self) -> usize {
        self.eta
    }
}

impl BfvPlanRequest {
    fn validate(&self) -> Result<(), BfvError> {
        let stages = self.widths.len();
        if stages == 0 ||
            self.slots.len() != stages ||
            self.slots.contains(&0) ||
            self.widths.iter().any(|w| !w.is_power_of_two() || *w > self.ring_dimension as usize) ||
            self.terminal_depth == 0 ||
            self.max_crt_depth < self.terminal_depth ||
            self.eta == 0
        {
            return Err(BfvError::Parameters("invalid public workload"));
        }
        DCRTPolyParams::validate_capability(
            self.ring_dimension,
            self.max_crt_depth,
            self.crt_bits,
            1,
        )
        .map_err(|_| BfvError::Parameters("unsupported prime pool"))?;
        Ok(())
    }
}

impl BfvParameterManifest {
    /// Restore the recorded bases, without generating a new prime pool, and
    /// verify every redundant certificate field before accepting the artifact.
    /// This verifies arithmetic consistency, not artifact authenticity.
    pub fn validate(&self) -> Result<BfvParameters, BfvError> {
        self.workload.validate()?;
        if self.ring_dimension != self.workload.ring_dimension ||
            self.eta != self.workload.eta ||
            self.levels.len() != self.workload.widths.len() + 1 ||
            self.levels
                .iter()
                .any(|base| base.is_empty() || base.len() > self.workload.max_crt_depth)
        {
            return Err(BfvError::Parameters("invalid manifest basis dimensions"));
        }
        let levels = self
            .levels
            .iter()
            .map(|base| {
                let bits =
                    base.iter().map(|p| (u64::BITS - p.leading_zeros()) as usize).max().unwrap();
                DCRTPolyParams::try_new(
                    self.ring_dimension,
                    base.len(),
                    bits,
                    1,
                    Some(base.clone()),
                    None,
                )
                .map_err(|_| BfvError::Parameters("invalid manifest prime basis"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = BfvParameters::new(levels, self.eta)?;
        if parameters.audit(&self.workload)? != *self {
            return Err(BfvError::Parameters(
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

    fn request() -> BfvPlanRequest {
        BfvPlanRequest {
            ring_dimension: std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
                .ok()
                .map(|value| value.parse().unwrap())
                .unwrap_or(16),
            widths: vec![1],
            slots: vec![1],
            eta: 1,
            crt_bits: 17,
            terminal_depth: 1,
            max_crt_depth: 16,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: BigUint::one(),
        }
    }

    #[test]
    fn test_bfv_manifest_roundtrip_and_certificate_validation() {
        let (parameters, manifest) = BfvParameters::plan(&request()).unwrap();
        let encoded = serde_json::to_vec(&manifest).unwrap();
        let decoded: BfvParameterManifest = serde_json::from_slice(&encoded).unwrap();
        let restored = decoded.validate().unwrap();
        assert_eq!(decoded, manifest);
        assert_eq!(restored.drops, parameters.drops);
        for (actual, expected) in restored.levels.iter().zip(&parameters.levels) {
            assert_eq!(actual.to_crt(), expected.to_crt());
            assert_eq!(actual.modulus(), expected.modulus());
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
        changed.eta += 1;
        invalid.push(changed);
        for (index, changed) in invalid.into_iter().enumerate() {
            assert!(changed.validate().is_err(), "accepted altered certificate {index}");
        }
    }

    #[test]
    fn test_bfv_planner_rejects_insufficient_public_budgets() {
        let base = request();
        let mut changed = base.clone();
        changed.max_crt_depth = changed.terminal_depth;
        assert!(matches!(BfvParameters::plan(&changed), Err(BfvError::Budget("CRT depth"))));
        let mut changed = base.clone();
        changed.memory_budget_bytes = 0;
        assert!(matches!(BfvParameters::plan(&changed), Err(BfvError::Budget("memory"))));
        let mut changed = base.clone();
        changed.terminal_value_bound = BigUint::from(1u8) << 128;
        assert!(matches!(BfvParameters::plan(&changed), Err(BfvError::Budget("terminal range"))));
        let mut changed = base;
        changed.slots[0] = usize::MAX;
        assert!(matches!(BfvParameters::plan(&changed), Err(BfvError::Budget("evaluation noise"))));
    }
}

#[derive(Clone)]
pub struct Blocks<const R: usize, const C: usize> {
    entries: [[Mat; C]; R],
    matrix_type: MatrixType,
}

pub type BfvRow = Blocks<1, 2>;

pub type BfvMatrix = Blocks<2, 2>;

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[[MatType; C]; R]: Serialize",
    deserialize = "[[MatType; C]; R]: Deserialize<'de>"
))]
pub struct BlocksType<const R: usize, const C: usize>([[MatType; C]; R]);

impl<const R: usize, const C: usize> BlocksType<R, C> {
    pub fn matrix_type(&self) -> MatrixType {
        let mut matrix_type = self.0[0][0].0.clone();
        matrix_type.rows = R.into();
        matrix_type.columns = C.into();
        matrix_type
    }
}

/// A structural product of branch fields, not a nested indexed family.
#[derive(Clone)]
pub struct BfvBranches(pub Vec<BfvMatrix>);

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvBranchesType(Vec<BlocksType<2, 2>>);

impl GraphValue for BfvBranches {
    type Schema = BfvBranchesType;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.0.iter().flat_map(GraphValue::flatten).collect()
    }
    fn schema(&self) -> Self::Schema {
        BfvBranchesType(self.0.iter().map(GraphValue::schema).collect())
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
                .map(|(schema, values)| BfvMatrix::from_values(schema, values))
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl GraphValueSchema for BfvBranchesType {
    type Value = BfvBranches;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BfvBranches(self.0.iter().map(|schema| schema.placeholders_from(next)).collect())
    }
    fn wire_types(&self) -> Vec<WireType> {
        self.0.iter().flat_map(GraphValueSchema::wire_types).collect()
    }
}

impl<const R: usize, const C: usize> Blocks<R, C> {
    pub fn new(entries: [[Mat; C]; R]) -> Result<Self, DslError> {
        if R == 0 || C == 0 {
            return Err(DslError::Schema);
        }
        let scalar_type = entries[0][0].matrix_type();
        if scalar_type.rows != 1.into() ||
            scalar_type.columns != 1.into() ||
            entries.iter().flatten().any(|entry| entry.matrix_type() != scalar_type)
        {
            return Err(DslError::Schema);
        }
        let mut matrix_type = scalar_type.clone();
        matrix_type.rows = R.into();
        matrix_type.columns = C.into();
        Ok(Self { entries, matrix_type })
    }

    pub fn entry(&self, row: usize, column: usize) -> &Mat {
        &self.entries[row][column]
    }
    pub fn matrix_type(&self) -> &MatrixType {
        &self.matrix_type
    }

    pub fn hash(ring: &Ring, key: Bytes, tag: impl Into<HashTag>) -> Self {
        let root = tag.into();
        Self::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let mut tag = root.clone();
                tag.push("bfv-block");
                tag.push(IntExpr::constant(row));
                tag.push(IntExpr::constant(column));
                ring.hash_matrix(key.clone(), tag, (1, 1))
            })
        }))
        .expect("one common scalar ring")
    }

    fn map(self, mut operation: impl FnMut(Mat) -> Mat) -> Self {
        Self::new(self.entries.map(|row| row.map(&mut operation)))
            .expect("block operation preserves scalar shape")
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
        self.map(|value| value.block_mod_switch(modulus.clone(), 1))
    }
}

impl BfvMatrix {
    pub fn from_rows(top: BfvRow, bottom: BfvRow) -> Self {
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

impl<const R: usize, const C: usize> GraphValue for Blocks<R, C> {
    type Schema = BlocksType<R, C>;
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
                        Mat::from_values(
                            &schema.0[row][column],
                            &values[row * C + column..row * C + column + 1],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .map_err(|_| DslError::Schema)
            })
            .collect::<Result<Vec<[Mat; C]>, _>>()?;
        Self::new(rows.try_into().map_err(|_| DslError::Schema)?)
    }
}

impl<const R: usize, const C: usize> GraphValueSchema for BlocksType<R, C> {
    type Value = Blocks<R, C>;
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
    fn test_bfv_polynomial_blocks_match_matrix_primitives() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let high = DCRTPolyParams::new(n, 3, 17, 1, None, None);
        let low = high.select_modulus(&high.to_crt().0[0].into()).unwrap();
        let high_modulus = BigInt::from(high.modulus().as_ref().clone());
        let low_modulus = BigInt::from(low.modulus().as_ref().clone());
        let ring = Ring::new(high_modulus.clone(), n as usize);
        let left = BfvMatrix::new(std::array::from_fn(|_| {
            std::array::from_fn(|_| ring.uniform_residue((1, 1)))
        }))
        .unwrap();
        let right = BfvMatrix::new(std::array::from_fn(|_| {
            std::array::from_fn(|_| ring.uniform_residue((1, 1)))
        }))
        .unwrap();
        // Concatenation is confined to this independent test oracle. Production
        // block arithmetic never reconstructs a dense matrix to slice it again.
        let dense = |blocks: &BfvMatrix| {
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
        let mut graph =
            DslContext::new("bfv-block-matrix-arithmetic").output("expected", expected).unwrap();
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
pub struct BfvPreparedLinear<K = BfvPublicKey, M = Mat> {
    lhs: K,
    rhs: K,
    output: K,
    operation: LinearOperation<M>,
}

impl<K, M> BfvPreparedLinear<K, M> {
    pub fn output_key(&self) -> &K {
        &self.output
    }
}

#[derive(Clone)]
pub struct BfvPayloadSecret {
    pub(crate) row: BfvRow,
    context: [u8; 32],
}

#[derive(Clone)]
pub struct BfvMaskSecret {
    pub(crate) row: BfvRow,
    context: [u8; 32],
}

/// Private payload-secret contribution prepared before the online mask secret
/// exists. This is not a public helper: it must never be exported publicly.
#[derive(Clone)]
pub struct BfvInputMaterial {
    pub(crate) row: BfvRow,
    level: usize,
    context: [u8; 32],
    key_id: [u8; 32],
}

#[cfg(test)]
pub(crate) struct TestSecrets {
    pub mask: BfvMaskSecret,
    pub payload: BfvPayloadSecret,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPublicKey<M = BfvMatrix> {
    pub(crate) matrix: M,
    pub(crate) level: usize,
    pub(crate) id: [u8; 32],
    pub(crate) context: [u8; 32],
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvEncoding<R = BfvRow> {
    pub(crate) row: R,
    pub(crate) level: usize,
    pub(crate) key_id: [u8; 32],
    pub(crate) context: [u8; 32],
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct MaskBranch<M = BfvMatrix> {
    sigma: usize,
    public: M,
}

#[derive(Clone)]
pub struct BfvMaskBank {
    id: [u8; 32],
    level: usize,
    context: [u8; 32],
    branches: Vec<MaskBranch>,
}

/// s-dependent rows prepared once per public mask bank and online secret.
/// The caller uses the same s for these rows and the input encodings.
#[derive(Clone, PartialEq)]
pub struct BfvOnlineMasks<R = BfvRow> {
    id: [u8; 32],
    level: usize,
    context: [u8; 32],
    rows: Arc<std::collections::BTreeMap<usize, R>>,
}

/// A fresh, fixed RHS and its mask-alignment material. `table=None` denotes
/// the identity function, and requires no registered identity table.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvLutMaterial<M = BfvMatrix> {
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
pub struct BfvRhsPlaintext {
    pub payload: Mat,
    pub id: [u8; 32],
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct PreparedBranch<M = BfvMatrix> {
    sigma: usize,
    // Share these graph values with public preparation. Constructing fresh
    // centered-extension nodes in evaluate_lut repeats the same conversion.
    lifted_ciphertext: M,
    ciphertext: M,
    // Family preparation exports the compact low-base intermediate and
    // recreates its lift inside the active online family body.
    intermediate: M,
    lifted_intermediate: M,
    mask_public: M,
}

impl PreparedBranch {
    fn from_parts(
        mask: &MaskBranch,
        ciphertext: BfvMatrix,
        intermediate: BfvMatrix,
        high: &BigInt,
        lifted_ciphertext: Option<&BfvMatrix>,
    ) -> Self {
        Self {
            sigma: mask.sigma,
            lifted_ciphertext: lifted_ciphertext
                .cloned()
                .unwrap_or_else(|| ciphertext.clone().centered_extend(high.clone())),
            ciphertext,
            lifted_intermediate: intermediate.clone().centered_extend(high.clone()),
            intermediate,
            mask_public: mask.public.clone(),
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPreparedLut<K = BfvPublicKey, M = BfvMatrix> {
    binding: BfvLutBinding,
    input: K,
    output: K,
    branches: Vec<PreparedBranch<M>>,
}

impl BfvPreparedLut {
    pub fn mask_bank(&self) -> BfvMaskBank {
        BfvMaskBank {
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
pub struct BfvScheme {
    parameter_identity: std::sync::OnceLock<[u8; 32]>,
    parameters: Arc<BfvParameters>,
    pub(crate) context: [u8; 32],
}

impl BfvPublicKey {
    pub fn matrix(&self) -> &BfvMatrix {
        &self.matrix
    }
    pub fn level(&self) -> usize {
        self.level
    }
    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }
}

impl BfvEncoding {
    pub fn row(&self) -> &BfvRow {
        &self.row
    }
    pub fn level(&self) -> usize {
        self.level
    }
}

impl BfvLutMaterial {
    pub fn table_commitment(&self) -> Option<&[u8; 32]> {
        self.table.as_ref()
    }
    pub fn has_rhs(&self) -> bool {
        self.rhs_id.is_some()
    }
}

impl BfvPreparedLut {
    pub fn binding(&self) -> &BfvLutBinding {
        &self.binding
    }
    pub fn output_key(&self) -> &BfvPublicKey {
        &self.output
    }
}

impl BfvScheme {
    pub fn new(parameters: BfvParameters, context: [u8; 32]) -> Self {
        Self {
            parameter_identity: std::sync::OnceLock::new(),
            parameters: Arc::new(parameters),
            context,
        }
    }
    pub fn parameters(&self) -> &BfvParameters {
        &self.parameters
    }
    pub fn ring(&self, level: usize) -> Result<Ring, BfvError> {
        let parameters = self.parameters.level(level)?;
        Ok(Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        ))
    }
    pub fn scalar(&self, level: usize, value: impl Into<BigInt>) -> Result<Mat, BfvError> {
        Ok(self.ring(level)?.polynomial([mxx_ir_core::IntExpr::constant(value.into())]))
    }
}

pub(self) fn sum<T: mxx_dsl::GraphValue + std::ops::Add<Output = T>>(
    family: Family<T>,
) -> Result<T, BfvError> {
    crate::utils::balanced_sum_family(family).map_err(|e| BfvError::Graph(e.to_string()))
}

/// A transition to the combined-output base, followed by extraction of one
/// CRT prime. The source is the preceding level; setup owns this contract.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct BfvRefreshLevel {
    pub output_level: usize,
    pub prime: BigInt,
}

/// Setup-owned output key and CRT idempotents in its full output ring.
pub struct BfvRefreshRecomposition {
    pub output_key: BfvPublicKey,
    pub coefficients: Vec<Mat>,
}

impl ExponentLUTRefreshScheme for BfvScheme {
    type RefreshRow = BfvRow;
    type RefreshDecoder = BfvRow;
    type RefreshLevel = BfvRefreshLevel;
    type RefreshRecomposition = BfvRefreshRecomposition;

    fn add_refresh_rows(&self, lhs: BfvRow, rhs: BfvRow) -> BfvRow {
        lhs + rhs
    }

    fn switch_refresh_row(&self, row: BfvRow, level: &BfvRefreshLevel) -> BfvRow {
        let q = self.parameters.level(level.output_level).expect("prepared output level").modulus();
        row.block_mod_switch(BigInt::from(q.as_ref().clone()))
    }

    fn apply_refresh_decoder(&self, switched: BfvRow, decoder: BfvRow) -> BfvRow {
        switched - decoder
    }

    fn extract_refresh_residue(&self, row: BfvRow, level: &BfvRefreshLevel) -> BfvRow {
        // Drop all primes except pi: P=H=q/pi. This is coefficient rounding,
        // not modular multiplication by H^-1 in the full output ring.
        row.block_mod_switch(level.prime.clone())
    }

    fn recompose_refresh_rows(
        &self,
        rows: Vec<BfvRow>,
        preparation: &BfvRefreshRecomposition,
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
        (BfvEncoding { row, level: key.level, key_id: key.id, context: self.context }, None)
    }
    type RefreshPublicRow = BfvMatrix;
    type RefreshTargetScale = Mat;

    fn add_refresh_public(&self, lhs: BfvMatrix, rhs: BfvMatrix) -> BfvMatrix {
        lhs + rhs
    }

    fn reduce_refresh_public(&self, value: BfvMatrix, level: &BfvRefreshLevel) -> BfvMatrix {
        let q = self.parameters.level(level.output_level).expect("prepared output level").modulus();
        value.reduce_modulus(BigInt::from(q.as_ref().clone()))
    }

    fn switch_refresh_public(&self, value: BfvMatrix, level: &BfvRefreshLevel) -> BfvMatrix {
        let q = self.parameters.level(level.output_level).expect("prepared output level").modulus();
        value.block_mod_switch(BigInt::from(q.as_ref().clone()))
    }

    fn scale_refresh_public(&self, value: BfvMatrix, scale: Mat) -> BfvMatrix {
        scale * value
    }

    fn subtract_refresh_public(&self, lhs: BfvMatrix, rhs: BfvMatrix) -> BfvMatrix {
        lhs - rhs
    }
}

impl ExponentLUTRefreshDirectScheme for BfvScheme {
    type RefreshArtifactBinding = BfvArtifactBinding;
    type RefreshOutputKey = BfvPublicKey;

    fn refresh_artifact_binding(&self) -> BfvArtifactBinding {
        self.artifact_binding()
    }

    fn refresh_output_key(preparation: &BfvRefreshRecomposition) -> &BfvPublicKey {
        &preparation.output_key
    }

    fn import_refresh_recomposition(
        &self,
        key: BfvPublicKey,
    ) -> Result<(BfvRefreshRecomposition, BlocksType<2, 2>, BlocksType<1, 2>), BfvError> {
        let ring = self.ring(key.level)?;
        let target = BlocksType(std::array::from_fn(|_| {
            std::array::from_fn(|_| MatType(ring.matrix_type((1, 1))))
        }));
        let offset = BlocksType(std::array::from_fn(|_| {
            std::array::from_fn(|_| MatType(ring.matrix_type((1, 1))))
        }));
        Ok((self.prepare_refresh_recomposition(key)?, target, offset))
    }

    type RefreshSecret = BfvMaskSecret;
    type RefreshDecoderParameters = usize;

    fn offset_refresh_decoder(&self, decoder: BfvRow, offset: BfvRow) -> BfvRow {
        decoder - offset
    }

    fn prepare_direct_refresh_decoder(
        &self,
        secret: &BfvMaskSecret,
        target: BfvMatrix,
        level: &BfvRefreshLevel,
        eta: &usize,
    ) -> BfvRow {
        let ring = self.ring(level.output_level).expect("prepared output level");
        let mask = self.mask_at(secret, level.output_level).expect("prepared secret context");
        if *eta == 0 {
            return mask * target;
        }
        let bits = mxx_dsl::parallel(*eta, |_| {
            BfvRow::new([std::array::from_fn(|_| {
                ring.uniform_interval((1, 1), 0, 1) - ring.uniform_interval((1, 1), 0, 1)
            })])
        })
        .expect("prepared decoder distribution");
        let error = crate::utils::balanced_sum_family(bits).expect("positive decoder eta");
        mask * target + error
    }
    type RefreshDigitSecret = BfvPayloadSecret;
    type RefreshDigitMaskBank = BfvMaskBank;

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
        (ProgramValue<Self::TerminalMaterial, Family<Self::TerminalMaterial>>, BfvRow),
        BfvError,
    > {
        use mxx_ir_core::{IntExpr, node::ConstantMatrix};
        let level = bank.level + 1;
        let ring = self.ring(level)?;
        let n = self.parameters.level(level)?.ring_dimension() as usize;
        let scale = self.scalar(level, routing.scale.clone())?;
        let weights = mxx_dsl::parallel(2 * n * routing.count, |index| {
            let (_, digit, coefficient, component) =
                crate::noise_refresh::refresh::route_coordinates(
                    index.expression()?,
                    2,
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
            let first =
                ring.polynomial([IntExpr::Sub(Box::new(1.into()), Box::new(component.clone()))]);
            let second = ring.polynomial([component]);
            BfvRow::new([[first * weight.clone(), second * weight]])
        })
        .map_err(|e| BfvError::Graph(e.to_string()))?;
        let maximum_weight =
            (1..routing.count).fold(routing.scale.clone(), |value, _| value * routing.base);
        let material = self.sample_readout_material(
            secret,
            readout,
            bank,
            hash_key,
            tag,
            id,
            Some(BfvReadoutWeights::Indexed { rows: weights, coefficient_bound: maximum_weight }),
            None,
        )?;
        let q = BigInt::from(self.parameters.level(level)?.modulus().as_ref().clone());
        let half = crate::utils::modular_inverse(&2.into(), &q).expect("odd output modulus");
        let offset = routing.offset_numerator() * half;
        let polynomial = ring.polynomial((0..n).map(|_| IntExpr::constant(offset.clone())));
        Ok((
            material.into_terminal(),
            BfvRow::new([[polynomial.clone(), polynomial]])
                .map_err(|e| BfvError::Graph(e.to_string()))?,
        ))
    }

    fn refresh_terminal_public(&self, raw: BfvTerminalAccumulator<BfvMatrix>) -> BfvMatrix {
        raw.value
    }

    fn refresh_terminal_encoding(&self, raw: BfvTerminalAccumulator<BfvRow>) -> BfvRow {
        raw.value
    }
}

/// Terminal stages use the same branch binding as ordinary LUTs, but their
/// output is a raw public accumulator at the input level. No finalized output
/// or s-dependent helper row is persisted here.
#[derive(Clone, Serialize, Deserialize)]
pub struct BfvPreparedTerminalDescriptor<P = BfvPreparedLutDescriptor> {
    prepared: P,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BfvArtifactBinding {
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
            BfvRow::new([[
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
            .add_outputs(DslContext::new("bfv-refresh-verification"), "residual", "decoded")
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
) -> Blocks<R, C> {
    BlocksType::<R, C>(std::array::from_fn(|_| {
        std::array::from_fn(|_| MatType(ring.matrix_type((1, 1))))
    }))
    .artifact_input(
        production.clone(),
        prefix,
        mxx_ir_core::artifact::ArtifactConfidentiality::Public,
    )
    .expect("one common scalar artifact ring")
}

/// Matrices contain only the active range; `range.start` identifies those
/// members in external RHS/public-value families of the full capacity.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPublicKeyFamily<M = ProgramValue<BfvMatrix, Family<BfvMatrix>>> {
    pub matrices: M,
    pub level: usize,
    pub id: [u8; 32],
    pub context: [u8; 32],
    pub range: FamilyRange,
}

#[derive(Clone)]
pub struct BfvEncodingFamily {
    pub rows: ProgramValue<BfvRow, Family<BfvRow>>,
    pub level: usize,
    pub key_id: [u8; 32],
    pub context: [u8; 32],
    pub range: FamilyRange,
}

pub struct BfvRhsPlaintextFamily {
    pub payloads: Family<Mat>,
    pub id: [u8; 32],
}

#[derive(Clone)]
pub struct BfvLutMaterialFamily {
    pub(self) binding: BfvLutBinding,
    /// Each member carries branch polynomial fields as a structural product.
    pub(self) ciphertexts: Family<BfvBranches>,
    pub(self) masks: Vec<MaskBranch>,
}

impl BfvLutMaterialFamily {
    /// Resolve one external member without expanding the structural family.
    pub(self) fn at(&self, index: Int) -> BfvLutMaterial {
        BfvLutMaterial {
            mask_bank: self.binding.mask_bank,
            id: self.binding.material_id,
            level: self.binding.level,
            context: self.binding.context,
            table: self.binding.table,
            rhs_id: self.binding.rhs,
            branches: self.masks.iter().cloned().zip(self.ciphertexts.at(index).0).collect(),
        }
    }

    pub fn binding(&self) -> &BfvLutBinding {
        &self.binding
    }
}

#[derive(Clone)]
pub struct BfvPreparedLutFamily {
    pub(self) input: BfvPublicKeyFamily,
    pub(self) output: BfvPublicKeyFamily,
    pub(self) material: BfvLutMaterialFamily,
    pub(self) intermediates: Family<BfvBranches>,
}

impl BfvPreparedLutFamily {
    pub fn mask_bank(&self) -> BfvMaskBank {
        BfvMaskBank {
            id: self.material.binding.mask_bank,
            level: self.input.level,
            context: self.input.context,
            branches: self.material.masks.clone(),
        }
    }

    pub fn output_key(&self) -> &BfvPublicKeyFamily {
        &self.output
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPreparedReduction<I = BfvPublicKeyFamily, K = BfvPublicKey> {
    pub(self) input: I,
    pub(self) output: K,
}

impl BfvPreparedReduction {
    pub fn output_key(&self) -> &BfvPublicKey {
        &self.output
    }
}

/// Public metadata used in place of encryption-time RHS plaintexts.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BfvLutBinding {
    pub mask_bank: [u8; 32],
    pub material_id: [u8; 32],
    pub level: usize,
    pub context: [u8; 32],
    pub table: Option<[u8; 32]>,
    pub rhs: Option<[u8; 32]>,
}

impl<M> BfvLutMaterial<M> {
    pub fn binding(&self) -> BfvLutBinding {
        BfvLutBinding {
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
pub struct BfvOnlineLutHelpers<R = BfvRow> {
    pub binding: BfvLutBinding,
    pub masks: crate::bfv::BfvOnlineMasks<R>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum PreparedGate<K = BfvPublicKey, M = BfvMatrix, P = Mat> {
    MulPrivate(BfvPreparedPrivateMul<K, M>),
    Linear(BfvPreparedLinear<K, P>),
    Lut(BfvPreparedLut<K, M>),
    /// Only the output key is needed to bind an online aggregate.
    Reduction(K),
}

#[derive(Clone, PartialEq)]
pub struct BfvProgramCache<G = PreparedGate, P = Mat> {
    pub(self) gates: BTreeMap<ProgramGateInstance, G>,
    pub(self) plaintexts: BTreeMap<ProgramGateInstance, P>,
}

// Gate instances are structured keys; JSON transports ordered entries, not object keys.
impl<G: Serialize, P: Serialize> Serialize for BfvProgramCache<G, P> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.gates.iter().collect::<Vec<_>>(), self.plaintexts.iter().collect::<Vec<_>>())
            .serialize(serializer)
    }
}
impl<'de, G: Deserialize<'de>, P: Deserialize<'de>> Deserialize<'de> for BfvProgramCache<G, P> {
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

impl<G, P> Default for BfvProgramCache<G, P> {
    fn default() -> Self {
        Self { gates: BTreeMap::new(), plaintexts: BTreeMap::new() }
    }
}

pub struct BfvPublicEvaluator<'a> {
    scheme: &'a BfvScheme,
    cache: &'a mut BfvProgramCache,
}

pub struct BfvEncodingEvaluator<'a> {
    scheme: &'a BfvScheme,
    cache: &'a BfvProgramCache,
}

impl ExponentLUTEncodingScheme for BfvScheme {
    fn reduce_public(
        &self,
        values: Family<Self::PublicKey>,
        gate: ProgramGateInstance,
        cache: &mut Self::Cache,
    ) -> Result<Self::PublicKey, BfvError> {
        let input = values.at(0).0;
        let matrix = crate::utils::balanced_sum_family(values.field(|(key, _)| key.matrix)?)
            .map_err(|e| BfvError::Graph(e.to_string()))?;
        let plaintext = if values.schema().element.1.is_some() {
            Some(
                crate::utils::balanced_sum_family(values.field(|(_, known)| known.unwrap())?)
                    .map_err(|e| BfvError::Graph(e.to_string()))?,
            )
        } else {
            None
        };
        let output = BfvPublicKey {
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
    ) -> Result<Self::Encoding, BfvError> {
        let Some(PreparedGate::Reduction(output)) = cache.gates.get(&gate) else {
            return Err(BfvError::Context);
        };
        let row = crate::utils::balanced_sum_family(values.field(|(value, _)| value.row)?)
            .map_err(|e| BfvError::Graph(e.to_string()))?;
        Ok((
            BfvEncoding { row, level: output.level, key_id: output.id, context: self.context },
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
    type PublicKey = (BfvPublicKey, Option<Mat>);
    type Encoding = (BfvEncoding, Option<Mat>);
    type PublicRhs = BfvPublicRhs;
    type EncodingRhs = BfvPublicRhs;
    type PublicHelpers = BfvProgramMaterial;
    type EncodingHelpers = BfvOnlineLutHelpers;
    type Cache = BfvProgramCache;
    type NoiseState = BfvNoiseValue;
    type NoiseModel<'a> = &'a crate::bfv::BfvParameters;
    type NoiseEvaluator<'a> = crate::bfv::noise::BfvNoiseEvaluator<'a>;
    fn noise_evaluator<'a>(
        _: &'a ExponentLutProgram,
        model: Self::NoiseModel<'a>,
        ranges: &'a ProgramFamilyBindings,
    ) -> Self::NoiseEvaluator<'a>
    where
        Self: 'a,
    {
        crate::bfv::noise::BfvNoiseEvaluator { parameters: model, ranges, gates: BTreeMap::new() }
    }
    fn reduce_noise<'a>(
        evaluator: &mut Self::NoiseEvaluator<'a>,
        value: &Self::NoiseState,
        range: &FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<Self::NoiseState, BfvError>
    where
        Self: 'a,
    {
        evaluator.reduce_add(value, range, gate)
    }
    type Error = BfvError;
    type PublicEvaluator<'a> = BfvPublicEvaluator<'a>;
    type EncodingEvaluator<'a> = BfvEncodingEvaluator<'a>;

    fn public_evaluator<'a>(
        &'a self,
        cache: &'a mut Self::Cache,
        _shared_cache: &'a Self::Cache,
        _ranges: &'a ProgramFamilyBindings,
    ) -> Self::PublicEvaluator<'a> {
        BfvPublicEvaluator { scheme: self, cache }
    }

    fn encoding_evaluator<'a>(
        &'a self,
        cache: &'a Self::Cache,
        _shared_cache: &'a Self::Cache,
        _ranges: &'a ProgramFamilyBindings,
    ) -> Self::EncodingEvaluator<'a> {
        BfvEncodingEvaluator { scheme: self, cache }
    }
}

pub(self) fn output_id(
    context: [u8; 32],
    operation: &[u8],
    gate: ProgramGateInstance,
    inputs: &[[u8; 32]],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"bfv-program-output");
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
    scheme: &BfvScheme,
    program: &ExponentLutProgram,
    lut: LutId,
    level: usize,
    rhs: Option<&[u8; 32]>,
    binding: &BfvLutBinding,
) -> Result<(), BfvError> {
    let table = program.lut(lut).ok_or(BfvError::Lut)?;
    if binding.context != scheme.context ||
        binding.level != level ||
        binding.table != Some(table.commitment()) ||
        binding.rhs.as_ref() != rhs
    {
        return Err(BfvError::Context);
    }
    let layout = program.layout().ok_or(BfvError::Lut)?;
    use mxx_primitives::poly::PolyParams;
    if layout.exponent.stride_log2 != 0 ||
        layout.ring_dimension_log2 !=
            scheme.parameters.level(level)?.ring_dimension().trailing_zeros()
    {
        return Err(BfvError::Parameters("incompatible BFV LUT layout"));
    }
    Ok(())
}

impl ProgramLoweringTypes for BfvPublicEvaluator<'_> {
    type Wire = (BfvPublicKey, Option<Mat>);
    type Rhs = BfvPublicRhs;
    type HelperSet = BfvProgramMaterial;
    type Error = BfvError;
}

impl ProgramLoweringTypes for BfvEncodingEvaluator<'_> {
    type Wire = (BfvEncoding, Option<Mat>);
    type Rhs = BfvPublicRhs;
    type HelperSet = BfvOnlineLutHelpers;
    type Error = BfvError;
}

pub type BfvPublicProgramValue =
    ProgramValue<(BfvPublicKey, Option<Mat>), Family<(BfvPublicKey, Option<Mat>)>>;
pub type BfvEncodingProgramValue =
    ProgramValue<(BfvEncoding, Option<Mat>), Family<(BfvEncoding, Option<Mat>)>>;

/// Fixed products consume a prepared ciphertext; fused LUT branches already
/// include the RHS plaintext, so their binding needs only its artifact identity.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum BfvPublicRhs<M = BfvMatrix> {
    Fixed(BfvPrivateRhs<M, M>),
    Lut([u8; 32]),
}
impl<M> BfvPublicRhs<M> {
    pub fn id(&self) -> [u8; 32] {
        match self {
            Self::Fixed(rhs) => rhs.rhs_id,
            Self::Lut(id) => *id,
        }
    }
}
pub type BfvProgramMaterial = BfvLutMaterial;

pub(self) fn lut_plaintext(
    scheme: &BfvScheme,
    program: &ExponentLutProgram,
    lut: LutId,
    level: usize,
    mu: &Mat,
) -> Result<Mat, BfvError> {
    use mxx_primitives::poly::PolyParams;
    let modulus =
        num_bigint::BigInt::from(scheme.parameters.level(level)?.modulus().as_ref().clone());
    let mu = mu.clone().reduce_modulus(modulus);
    let table = program.lut(lut).ok_or(BfvError::Lut)?;
    let layout = crate::layout::ExpLayout {
        stride_log2: 0,
        width_log2: table.values().len().trailing_zeros(),
    };
    let ring = scheme.ring(level)?;
    let sigmas = layout
        .automorphism_indices(scheme.parameters.level(level)?.ring_dimension() as usize)
        .map_err(|_| BfvError::Lut)?;
    let terms = sigmas
        .into_iter()
        .map(|sigma| {
            let coefficient = crate::utils::lut_coefficient(&ring, &layout, sigma, table.values())
                .map_err(|_| BfvError::Lut)?;
            Ok(coefficient * mu.clone().ring_automorphism(sigma))
        })
        .collect::<Result<Vec<_>, BfvError>>()?;
    crate::utils::balanced_sum_family(
        mxx_dsl::Family::pack(terms).map_err(|e| BfvError::Graph(e.to_string()))?,
    )
    .map_err(|e| BfvError::Graph(e.to_string()))
}

/// Public row-polynomial weight for coefficient routing and digit scaling.
/// Empty coefficient lists represent zero polynomials.
pub type BfvReadoutWeight = [Vec<BigInt>; 2];

/// Public weights at the terminal output level. Indexed bounds are setup
/// contracts; no ciphertext coefficient scan is performed. Runtime values retain
/// the indexed rows; serialized metadata uses `()` because those rows are stored
/// as public graph artifacts.
#[derive(Clone, Serialize, Deserialize)]
pub enum BfvReadoutWeights<R = Family<BfvRow>> {
    Shared(BfvReadoutWeight),
    Indexed { rows: R, coefficient_bound: BigUint },
}

/// Public preprocessing ports, bound to a readout and terminal stage.
#[derive(Clone, Serialize, Deserialize)]
pub struct BfvPreparedReadoutDescriptor {
    prefix: String,
    identity: ValueReadoutIdentity,
    terminal: ProgramValue<BfvPreparedTerminalDescriptor, BfvPreparedTerminalFamilyDescriptor>,
    weight: BfvReadoutWeights<()>,
}

pub struct BfvReadoutMaterial {
    mode: BfvTerminalFamilyMode,
    identity: ValueReadoutIdentity,
    terminal: ProgramValue<BfvTerminalMaterial, Family<BfvTerminalMaterial>>,
    target_bound: BigUint,
    weight: BfvReadoutWeights,
}

impl BfvReadoutMaterial {
    /// Raw targets already include the inverse-two normalization. A refresh
    /// aggregate supplies its own exact integer-distribution offset.
    pub fn into_terminal(self) -> ProgramValue<BfvTerminalMaterial, Family<BfvTerminalMaterial>> {
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
pub struct BfvPreparedReadout {
    identity: ValueReadoutIdentity,
    public: BfvReadoutPublicOutput,
    terminal: BfvReadoutTerminal,
}

enum BfvReadoutTerminal {
    Scalar { terminal: BfvPreparedTerminal, offset: BfvRow },
    Family { terminal: BfvPreparedTerminalFamily, offset: ProgramValue<BfvRow, Family<BfvRow>> },
}

impl BfvPreparedReadout {
    pub fn mask_bank(&self) -> BfvMaskBank {
        match &self.terminal {
            BfvReadoutTerminal::Scalar { terminal, .. } => terminal.mask_bank(),
            BfvReadoutTerminal::Family { terminal, .. } => terminal.mask_bank(),
        }
    }
    pub fn binding(&self) -> crate::bfv::BfvLutBinding {
        match &self.terminal {
            BfvReadoutTerminal::Scalar { terminal, .. } => terminal.binding().clone(),
            BfvReadoutTerminal::Family { terminal, .. } => terminal.binding(),
        }
    }
    pub fn id(&self) -> [u8; 32] {
        match &self.terminal {
            BfvReadoutTerminal::Scalar { terminal, .. } => terminal.id(),
            BfvReadoutTerminal::Family { terminal, .. } => terminal.id(),
        }
    }
}

#[derive(Clone)]
pub struct BfvReadoutPublicOutput {
    pub values: ProgramValue<BfvMatrix, (Family<BfvMatrix>, FamilyRange)>,
    level: usize,
    weight: BfvReadoutWeights,
}

impl BfvReadoutPublicOutput {
    pub fn weight(&self) -> &BfvReadoutWeights {
        &self.weight
    }

    pub fn level(&self) -> usize {
        self.level
    }
}

pub struct BfvReadoutEncodingOutput {
    pub values: ProgramValue<BfvRow, (Family<BfvRow>, FamilyRange)>,
    level: usize,
}

impl BfvReadoutEncodingOutput {
    pub fn level(&self) -> usize {
        self.level
    }
}

pub(self) fn readout_offset(
    ring: &mxx_dsl::Ring,
    weight: &BfvReadoutWeight,
) -> Result<BfvRow, BfvError> {
    BfvRow::new([weight.each_ref().map(|polynomial| {
        if polynomial.is_empty() {
            ring.zero((1, 1))
        } else {
            ring.polynomial(polynomial.iter().cloned().map(Into::into))
        }
    })])
    .map_err(|e| BfvError::Graph(e.to_string()))
}

impl ExponentLUTReadoutScheme for BfvScheme {
    type ReadoutArtifacts = BfvPreparedReadout;
    type ReadoutPreparation = BfvReadoutMaterial;
    type ReadoutPublicMasks = ();
    type ReadoutEncodingMasks = BfvOnlineMasks;
    type ReadoutPublicOutput = BfvReadoutPublicOutput;
    type ReadoutEncodingOutput = BfvReadoutEncodingOutput;
    type ReadoutError = BfvError;

    fn prepare_readout(
        &self,
        readout: &ValueReadout,
        inputs: &BfvPublicProgramValue,
        preparation: &BfvReadoutMaterial,
    ) -> Result<BfvPreparedReadout, BfvError> {
        if preparation.identity != readout.identity() {
            return Err(BfvError::Context);
        }
        let level = match inputs {
            BfvPublicProgramValue::Scalar((input, _)) => input.level + 1,
            BfvPublicProgramValue::Family(keys) => keys.schema().element.0.level + 1,
        };
        let modulus = BigInt::from(self.parameters.level(level)?.modulus().as_ref().clone());
        let half = self.scalar(
            level,
            crate::utils::modular_inverse(&2.into(), &modulus).ok_or(BfvError::Lut)?,
        )?;
        let ring = self.ring(level)?;
        let offset = match &preparation.weight {
            BfvReadoutWeights::Shared(weight) => {
                ProgramValue::Scalar(half.clone() * readout_offset(&ring, weight)?)
            }
            BfvReadoutWeights::Indexed { rows, .. } => ProgramValue::Family(
                mxx_dsl::parallel(rows.count().clone(), |index| Ok(half.clone() * rows.at(index)))
                    .map_err(|e| BfvError::Graph(e.to_string()))?,
            ),
        };
        match inputs {
            BfvPublicProgramValue::Scalar((input, _)) => {
                let ProgramValue::Scalar(material) = &preparation.terminal else {
                    return Err(BfvError::Context);
                };
                let ProgramValue::Scalar(offset) = offset else {
                    return Err(BfvError::Context);
                };
                let terminal = self.prepare_terminal(&input, material)?;
                let terminal_public =
                    self.finalize_terminal_public(terminal.public_accumulator())?;
                let public = BfvReadoutPublicOutput {
                    values: ProgramValue::Scalar(terminal_public.matrix().clone()),
                    level,
                    weight: preparation.weight.clone(),
                };
                Ok(BfvPreparedReadout {
                    identity: preparation.identity,
                    public,
                    terminal: BfvReadoutTerminal::Scalar { terminal, offset },
                })
            }
            BfvPublicProgramValue::Family(_) => {
                let key = BfvPublicKeyFamily::from_program(inputs, 1.into())?;
                let terminal = self.prepare_terminal_family(
                    &key,
                    &preparation.terminal,
                    preparation.mode,
                    None,
                )?;
                let offset = if preparation.mode == BfvTerminalFamilyMode::Sum {
                    ProgramValue::Scalar(match offset {
                        ProgramValue::Scalar(row) => {
                            ring.polynomial([key.range.count().clone()]) * row
                        }
                        ProgramValue::Family(rows) => crate::utils::balanced_sum_family(rows)
                            .map_err(|e| BfvError::Graph(e.to_string()))?,
                    })
                } else {
                    offset
                };
                let matrices = terminal.public_matrices().clone();
                let public = BfvReadoutPublicOutput {
                    values: ProgramValue::Family((matrices, terminal.output_range())),
                    level,
                    weight: preparation.weight.clone(),
                };
                Ok(BfvPreparedReadout {
                    identity: preparation.identity,
                    public,
                    terminal: BfvReadoutTerminal::Family { terminal, offset },
                })
            }
        }
    }

    fn project_readout(
        &self,
        readout: &ValueReadout,
        inputs: &BfvPublicProgramValue,
        artifacts: &BfvPreparedReadout,
        _masks: &(),
    ) -> Result<BfvReadoutPublicOutput, BfvError> {
        match (inputs, &artifacts.terminal) {
            (
                BfvPublicProgramValue::Scalar((input, _)),
                BfvReadoutTerminal::Scalar { terminal, .. },
            ) => {
                self.check_key(&input)?;
                if artifacts.identity != readout.identity() ||
                    input.id != terminal.input.id ||
                    input.level != terminal.input.level ||
                    input.context != terminal.input.context
                {
                    return Err(BfvError::Context);
                }
                Ok(artifacts.public.clone())
            }
            (BfvPublicProgramValue::Family(_), BfvReadoutTerminal::Family { terminal, .. }) => {
                let key = BfvPublicKeyFamily::from_program(inputs, 1.into())?;
                self.check_public_family(&key)?;
                let expected = terminal.input_key();
                if artifacts.identity != readout.identity() ||
                    key.id != expected.id ||
                    key.level != expected.level ||
                    key.context != expected.context ||
                    key.range != expected.range
                {
                    return Err(BfvError::Context);
                }
                Ok(artifacts.public.clone())
            }
            _ => Err(BfvError::Context),
        }
    }

    fn evaluate_readout(
        &self,
        readout: &ValueReadout,
        inputs: &BfvEncodingProgramValue,
        artifacts: &BfvPreparedReadout,
        masks: &BfvOnlineMasks,
    ) -> Result<BfvReadoutEncodingOutput, BfvError> {
        match (inputs, &artifacts.terminal) {
            (
                BfvEncodingProgramValue::Scalar((input, _)),
                BfvReadoutTerminal::Scalar { terminal, offset },
            ) => {
                if artifacts.identity != readout.identity() {
                    return Err(BfvError::Context);
                }
                let raw = self.evaluate_terminal(input, terminal, masks)?;
                // The finalized matrix is already owned by the public readout output.
                // Reuse that handle with the terminal binding; no public arithmetic
                // or additional artifact is needed during online evaluation.
                let terminal_public = BfvTerminalPublicKey {
                    matrix: artifacts.public.values.scalar().ok_or(BfvError::Context)?.clone(),
                    level: artifacts.public.level,
                    context: terminal.input.context,
                    stages: vec![terminal.id],
                };
                let terminal = self.finalize_terminal_encoding(raw, &terminal_public)?;
                Ok(BfvReadoutEncodingOutput {
                    values: ProgramValue::Scalar(terminal.row().clone() + offset.clone()),
                    level: terminal.level(),
                })
            }
            (
                BfvEncodingProgramValue::Family(_),
                BfvReadoutTerminal::Family { terminal, offset },
            ) => {
                let encoding = BfvEncodingFamily::from_program(inputs, 1.into())?;
                if artifacts.identity != readout.identity() {
                    return Err(BfvError::Context);
                }
                let evaluated = self.evaluate_terminal_family(&encoding, terminal, masks)?;
                let rows = mxx_dsl::parallel(evaluated.count().clone(), |index| {
                    Ok(evaluated.at(&index) + offset.at(index))
                })
                .map_err(|e| BfvError::Graph(e.to_string()))?;
                Ok(BfvReadoutEncodingOutput {
                    values: ProgramValue::Family((rows, terminal.output_range())),
                    level: terminal.level(),
                })
            }
            _ => Err(BfvError::Context),
        }
    }
}

/// A terminal target, distinct from ordinary state material. Shared and indexed
/// storage use the same marker; construction remains inside this backend.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvTerminalMaterial<M = BfvLutMaterial, L = Vec<BfvMatrix>> {
    pub(self) material: M,
    pub(self) lifted_ciphertexts: L,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvPreparedTerminal<K = BfvPublicKey, M = BfvMatrix, P = M> {
    pub(self) binding: BfvLutBinding,
    pub(self) input: K,
    pub(self) id: [u8; 32],
    pub(self) branches: Vec<PreparedBranch<M>>,
    pub(self) raw_public: P,
}

/// All terms stay at their common input base. Only setup identities grow with
/// the number of terms; a carrier holds one polynomial-block matrix or row.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BfvTerminalAccumulator<T> {
    pub(self) value: T,
    pub(self) level: usize,
    pub(self) context: [u8; 32],
    pub(self) stages: Vec<[u8; 32]>,
}

pub struct BfvTerminalPublicKey {
    pub(self) matrix: BfvMatrix,
    pub(self) level: usize,
    pub(self) context: [u8; 32],
    pub(self) stages: Vec<[u8; 32]>,
}

pub struct BfvTerminalEncoding {
    row: BfvRow,
    level: usize,
}

impl BfvTerminalPublicKey {
    pub fn matrix(&self) -> &BfvMatrix {
        &self.matrix
    }
    pub fn level(&self) -> usize {
        self.level
    }
}

impl BfvTerminalEncoding {
    pub fn row(&self) -> &BfvRow {
        &self.row
    }
    pub fn level(&self) -> usize {
        self.level
    }
}

impl<P> BfvPreparedTerminal<BfvPublicKey, BfvMatrix, P> {
    pub fn binding(&self) -> &BfvLutBinding {
        &self.binding
    }
    pub fn id(&self) -> [u8; 32] {
        self.id
    }
    /// Public masks suffice to generate fresh helpers for another online s.
    pub fn mask_bank(&self) -> BfvMaskBank {
        BfvMaskBank {
            id: self.binding.mask_bank,
            level: self.input.level,
            context: self.input.context,
            branches: self
                .branches
                .iter()
                .map(|branch| crate::bfv::MaskBranch {
                    sigma: branch.sigma,
                    public: branch.mask_public.clone(),
                })
                .collect(),
        }
    }
}

impl BfvPreparedTerminal {
    pub fn public_accumulator(&self) -> BfvTerminalAccumulator<BfvMatrix> {
        BfvTerminalAccumulator {
            value: self.raw_public.clone(),
            level: self.input.level,
            context: self.input.context,
            stages: vec![self.id],
        }
    }
}

impl<T: std::ops::Add<Output = T>> BfvTerminalAccumulator<T> {
    pub fn combine(mut self, other: Self) -> Result<Self, BfvError> {
        if self.level != other.level || self.context != other.context {
            return Err(BfvError::Context);
        }
        self.value = self.value + other.value;
        self.stages.extend(other.stages);
        Ok(self)
    }
}

impl ExponentLUTTerminalScheme for BfvScheme {
    fn reduce_terminal_public(
        &self,
        values: Family<Self::TerminalPublicAccumulator>,
    ) -> Result<Self::TerminalPublicAccumulator, BfvError> {
        let metadata = values.schema().element;
        let value = crate::utils::balanced_sum_family(values.field(|value| value.value)?)
            .map_err(|e| BfvError::Graph(e.to_string()))?;
        Ok(BfvTerminalAccumulator {
            value,
            level: metadata.level,
            context: metadata.context,
            stages: metadata.stages,
        })
    }
    fn reduce_terminal_encoding(
        &self,
        values: Family<Self::TerminalEncodingAccumulator>,
    ) -> Result<Self::TerminalEncodingAccumulator, BfvError> {
        let metadata = values.schema().element;
        let value = crate::utils::balanced_sum_family(values.field(|value| value.value)?)
            .map_err(|e| BfvError::Graph(e.to_string()))?;
        Ok(BfvTerminalAccumulator {
            value,
            level: metadata.level,
            context: metadata.context,
            stages: metadata.stages,
        })
    }

    type TerminalMaterial = BfvTerminalMaterial;
    type TerminalOnlineHelpers = BfvOnlineMasks;
    type PreparedTerminal = BfvPreparedTerminal<BfvPublicKey, BfvMatrix, ()>;
    type TerminalPublicAccumulator = BfvTerminalAccumulator<BfvMatrix>;
    type TerminalEncodingAccumulator = BfvTerminalAccumulator<BfvRow>;
    type TerminalPublicOutput = BfvTerminalPublicKey;
    type TerminalEncodingOutput = BfvTerminalEncoding;
    type TerminalDescriptor =
        BfvPreparedTerminalDescriptor<(String, <Self::PreparedTerminal as GraphValue>::Schema)>;
    type TerminalBinding = (BfvLutBinding, [u8; 32], [u8; 32]);

    fn export_terminal_term(
        &self,
        term: &Self::PreparedTerminal,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, Self::TerminalDescriptor), BfvError> {
        Ok((
            graph.public_output(prefix, term.clone())?,
            BfvPreparedTerminalDescriptor { prepared: (prefix.to_owned(), term.schema()) },
        ))
    }
    fn import_terminal_term(
        &self,
        descriptor: &Self::TerminalDescriptor,
        production: ProductionId,
        expected: &Self::TerminalBinding,
    ) -> Result<Self::PreparedTerminal, BfvError> {
        let (prefix, schema) = &descriptor.prepared;
        if schema.binding != expected.0 ||
            schema.input.id != expected.1 ||
            schema.id != expected.2 ||
            schema.input.context != self.context
        {
            return Err(BfvError::Context);
        }
        let term = schema.artifact_input(production, prefix, ArtifactConfidentiality::Public)?;
        self.check_key(&term.input)?;
        Ok(term)
    }
    fn export_terminal_output(
        &self,
        public: &BfvTerminalPublicKey,
        graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<mxx_dsl::DslContext, BfvError> {
        Ok(graph.public_output(prefix, public.matrix.clone())?)
    }
    fn import_terminal_output(
        &self,
        terms: &[Self::PreparedTerminal],
        production: mxx_ir_core::artifact::ProductionId,
        prefix: &str,
    ) -> Result<BfvTerminalPublicKey, BfvError> {
        let input_level = terms[0].input.level;
        if terms.iter().any(|term| term.input.level != input_level) {
            return Err(BfvError::Context);
        }
        let level = input_level + 1;
        Ok(BfvTerminalPublicKey {
            matrix: crate::bfv::import_blocks(&self.ring(level)?, &production, prefix),
            level,
            context: self.context,
            stages: terms.iter().map(|term| term.id).collect(),
        })
    }

    fn prepare_terminal_term(
        &self,
        input: &Self::PublicKey,
        material: &Self::TerminalMaterial,
    ) -> Result<(Self::PreparedTerminal, Self::TerminalPublicAccumulator), BfvError> {
        let prepared = self.prepare_terminal(&input.0, material)?;
        let raw = BfvTerminalAccumulator {
            value: prepared.raw_public,
            level: prepared.input.level,
            context: prepared.input.context,
            stages: vec![prepared.id],
        };
        let cache = BfvPreparedTerminal {
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
    ) -> Result<Self::TerminalEncodingAccumulator, BfvError> {
        self.evaluate_terminal(&input.0, prepared, helpers)
    }

    fn add_terminal_public(
        &self,
        lhs: Self::TerminalPublicAccumulator,
        rhs: Self::TerminalPublicAccumulator,
    ) -> Result<Self::TerminalPublicAccumulator, BfvError> {
        lhs.combine(rhs)
    }

    fn add_terminal_encoding(
        &self,
        lhs: Self::TerminalEncodingAccumulator,
        rhs: Self::TerminalEncodingAccumulator,
    ) -> Result<Self::TerminalEncodingAccumulator, BfvError> {
        lhs.combine(rhs)
    }

    fn finish_terminal_public(
        &self,
        raw: Self::TerminalPublicAccumulator,
    ) -> Result<Self::TerminalPublicOutput, BfvError> {
        self.finalize_terminal_public(raw)
    }

    fn finish_terminal_encoding(
        &self,
        raw: Self::TerminalEncodingAccumulator,
        public: &Self::TerminalPublicOutput,
    ) -> Result<Self::TerminalEncodingOutput, BfvError> {
        self.finalize_terminal_encoding(raw, public)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BfvTerminalFamilyMode {
    /// One separately finalized output per input member.
    Independent,
    /// One output containing the sum of raw members, finalized once.
    Sum,
}

pub struct BfvPreparedTerminalFamily {
    mode: BfvTerminalFamilyMode,
    finalize: bool,
    input: BfvPublicKeyFamily,
    id: [u8; 32],
    cache: Family<BfvPreparedTerminal<BfvPublicKey, BfvMatrix, ()>>,
    public: Family<BfvMatrix>,
}

impl BfvPreparedTerminalFamily {
    pub fn input_key(&self) -> &BfvPublicKeyFamily {
        &self.input
    }
    pub fn binding(&self) -> crate::bfv::BfvLutBinding {
        self.cache.schema().element.binding
    }
    pub fn id(&self) -> [u8; 32] {
        self.id
    }

    pub fn public_matrices(&self) -> &Family<BfvMatrix> {
        &self.public
    }

    pub fn level(&self) -> usize {
        self.input.level + usize::from(self.finalize)
    }

    pub fn mode(&self) -> BfvTerminalFamilyMode {
        self.mode
    }

    /// Input range; aggregate mode has a singleton output family.
    pub fn range(&self) -> &crate::program::FamilyRange {
        &self.input.range
    }

    /// Output range preserves independent members or names the singleton sum.
    pub fn output_range(&self) -> crate::program::FamilyRange {
        match self.mode {
            BfvTerminalFamilyMode::Independent => self.input.range.clone(),
            BfvTerminalFamilyMode::Sum => {
                crate::program::FamilyRange::bounded(0, 1, 1).expect("singleton range")
            }
        }
    }

    pub fn mask_bank(&self) -> BfvMaskBank {
        self.cache.at(0).mask_bank()
    }
}

/// Structural terminal ports and shared/indexed public material, bound at setup.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BfvPreparedTerminalFamilyDescriptor {
    prefix: String,
    parameters: [u8; 32],
    mode: BfvTerminalFamilyMode,
    finalize: bool,
    id: [u8; 32],
    range: FamilyRange,
    cache: <Family<BfvPreparedTerminal<BfvPublicKey, BfvMatrix, ()>> as GraphValue>::Schema,
    public: FamilyType<BlocksType<2, 2>>,
}
