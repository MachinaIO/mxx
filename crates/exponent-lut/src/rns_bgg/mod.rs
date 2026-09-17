//! The shared, scheme-independent RNS-BGG core.
//!
//! This module intentionally contains only the contracts shared by the stage
//! owners. Public-key preparation, encoding/encryption, and noise evaluation
//! live in their respective modules; this file does not provide a scheme
//! facade or an artifact/material wrapper hierarchy.

mod encoding;
#[cfg(all(test, feature = "gpu"))]
pub(crate) mod gpu;
mod noise;
mod public_key;

use bigdecimal::BigDecimal;
use mxx_dsl::{Family, Mat, Preimage, Ring};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::{
    noise_refresh::{
        prf::PrfError,
        refresh::{
            RefreshDeclaration, RefreshDigitPlan, RefreshError, RefreshPrfLabelIndex, RefreshSlot,
        },
    },
    program::{ExponentLUTProgramLowering, ExponentLutProgramId, FamilyScope, WireId},
};

pub use encoding::{
    EncodingLowering, RnsBggEncoding, RnsBggMaskSecret, RnsBggPayloadSecret, RnsEncryptBuild,
    RnsOnlineBuild,
};
pub use noise::{
    RnsBggDerivedWorkload, RnsBggMaskSecretComponent, RnsBggNoiseBindings, RnsBggNoiseBound,
    RnsBggNoiseEvaluator, RnsBggNoiseFamily, RnsBggNoiseRhs, RnsBggNoiseTrace, RnsBggNoiseValue,
    RnsBggParameterManifest, RnsBggPlanRequest, RnsNoiseBuild,
};
pub use public_key::{PublicKeyLowering, RnsBggPublicKey, RnsPreprocessBuild};

/// A two-polynomial row block.
pub type RnsBggRow = (Mat, Mat);
/// A two-by-two polynomial block matrix.
pub type RnsBggMatrix = ((Mat, Mat), (Mat, Mat));
/// A two-by-two matrix of signed compact preimages.
pub type RnsBggDecomposition = ((Preimage, Preimage), (Preimage, Preimage));
/// Public LUT data indexed by the LUT helper family.
pub type RnsBggLutPublicData = Family<RnsBggMatrix>;
/// LUT helper rows indexed by the LUT helper family.
pub type RnsBggLutHelper = Family<RnsBggRow>;

fn refresh_commitment_hex(commitment: [u8; 32]) -> String {
    commitment.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Build the canonical per-slot refresh artifact path.
pub(crate) fn refresh_artifact_name(commitment: [u8; 32], slot: usize, role: &str) -> String {
    format!("refresh/{}/slot/{slot}/{role}", refresh_commitment_hex(commitment))
}

/// Build the canonical per-slot prepared-PRF artifact path.
pub(crate) fn refresh_prf_artifact_name(commitment: [u8; 32], slot: usize, role: &str) -> String {
    refresh_artifact_name(commitment, slot, &format!("prf/{role}"))
}

/// Build the canonical label-specific target-action artifact path.
///
/// Label actions are currently target actions; the fixed role is part of this
/// helper so producer and consumer cannot accidentally diverge.
pub(crate) fn refresh_label_artifact_name(
    commitment: [u8; 32],
    slot: usize,
    label: usize,
) -> String {
    refresh_artifact_name(commitment, slot, &format!("target/{label}"))
}

/// Build the canonical refresh-checkpoint artifact path.
pub(crate) fn refresh_checkpoint_artifact_name(commitment: [u8; 32], role: &str) -> String {
    format!("refresh/{}/{role}", refresh_commitment_hex(commitment))
}

/// Build the canonical base name for a LUT helper artifact.
///
/// A root helper keeps the historical name.  A helper produced in a family
/// scope carries every lexical range's start and count so nested scopes that
/// happen to share an inner index remain distinct.
pub(crate) fn lut_helper_artifact_name(origin: WireId, scope: &FamilyScope) -> String {
    if scope.ranges().is_empty() {
        return format!("lut-data/{origin:?}");
    }
    let ranges = scope
        .ranges()
        .iter()
        .map(|range| format!("start={:?},count={:?}", range.start(), range.count()))
        .collect::<Vec<_>>()
        .join("/");
    format!("lut-data/{origin:?}/scope/{ranges}")
}

/// Errors raised by the retained RNS-BGG contracts.
#[derive(Debug, Error)]
pub enum RnsBggError {
    #[error("invalid RNS-BGG parameters: {0}")]
    Parameters(&'static str),
    #[error("RNS-BGG budget exceeded: {0}")]
    Budget(&'static str),
    #[error("invalid RNS-BGG artifact or graph binding: {0}")]
    Binding(&'static str),
    #[error("RHS {rhs:?} is consumed at level {expected}, not {actual}")]
    RhsLevelMismatch { rhs: crate::program::RhsId, expected: usize, actual: usize },
    #[error(transparent)]
    Prf(#[from] PrfError),
    #[error(transparent)]
    Refresh(#[from] crate::noise_refresh::refresh::RefreshError),
    #[error("invalid RNS-BGG LUT")]
    Lut,
    #[error("RNS-BGG graph construction failed: {0}")]
    Graph(String),
}

impl From<mxx_dsl::DslError> for RnsBggError {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

fn canonical_encoding_sigma(sigma: BigDecimal) -> Result<BigDecimal, RnsBggError> {
    let sigma = sigma.normalized();
    let (_, exponent) = sigma.as_bigint_and_exponent();
    if sigma <= BigDecimal::zero() || exponent.unsigned_abs() > u64::from(u32::MAX) {
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

/// Validated modulus-chain and encoding-noise parameters.
#[derive(Clone, Debug)]
pub struct RnsBggParameters {
    levels: Vec<DCRTPolyParams>,
    drops: Vec<BigUint>,
    encoding_error_sigma: BigDecimal,
    encoding_error_bound: BigUint,
    operations_per_drop: usize,
}

impl RnsBggParameters {
    /// Validates the ring chain and computes every modulus-drop scale once.
    pub fn new(
        levels: Vec<DCRTPolyParams>,
        sigma: impl Into<BigDecimal>,
        operations_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        if levels.is_empty() {
            return Err(RnsBggError::Parameters("empty modulus chain"));
        }
        if operations_per_drop == 0 {
            return Err(RnsBggError::Parameters("operations_per_drop must be positive"));
        }
        let sigma = canonical_encoding_sigma(sigma.into())?;
        let encoding_error_bound = encoding_error_bound(&sigma)?;
        let first = &levels[0];
        if first.dropped_moduli() != 0 {
            return Err(RnsBggError::Parameters("partial modulus bases are not supported"));
        }
        if levels.iter().any(|level| {
            level.ring_dimension() != first.ring_dimension() || level.dropped_moduli() != 0
        }) {
            return Err(RnsBggError::Parameters("incompatible ring or partial basis"));
        }

        let mut drops = Vec::with_capacity(levels.len());
        for pair in levels.windows(2) {
            if pair[0] == pair[1] {
                drops.push(BigUint::from(1u8));
                continue;
            }
            let source = pair[0].to_crt().0;
            let target = pair[1].to_crt().0;
            if target.len() >= source.len() || target.iter().any(|prime| !source.contains(prime)) {
                return Err(RnsBggError::Parameters(
                    "levels must be equal or strictly nested prime bases",
                ));
            }
            let source_modulus = pair[0].modulus();
            let target_modulus = pair[1].modulus();
            let quotient = source_modulus.as_ref() / target_modulus.as_ref();
            if &quotient * target_modulus.as_ref() != *source_modulus.as_ref() {
                return Err(RnsBggError::Parameters("modulus drop is not exact"));
            }
            drops.push(quotient);
        }
        drops.push(BigUint::from(1u8));
        Ok(Self {
            levels,
            drops,
            encoding_error_sigma: sigma,
            encoding_error_bound,
            operations_per_drop,
        })
    }

    pub fn levels(&self) -> &[DCRTPolyParams] {
        &self.levels
    }

    pub fn level(&self, level: usize) -> Result<&DCRTPolyParams, RnsBggError> {
        self.levels.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }

    pub fn scale(&self, level: usize) -> Result<&BigUint, RnsBggError> {
        self.drops.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }

    pub fn component_columns(&self, level: usize) -> Result<usize, RnsBggError> {
        self.level(level)?;
        Ok(self.levels.get(level + 1).map_or(1, PolyParams::modulus_digits))
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

    pub fn ring(&self, level: usize) -> Result<Ring, RnsBggError> {
        let parameters = self.level(level)?;
        Ok(Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        ))
    }

    pub fn scalar(&self, level: usize, value: impl Into<BigInt>) -> Result<Mat, RnsBggError> {
        Ok(self.ring(level)?.polynomial([value.into().into()]))
    }

    /// Stable identity of the validated parameter contract.
    pub(crate) fn parameter_identity(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-rns-bgg-parameters/v1");
        digest.update(self.encoding_error_sigma.normalized().to_string().as_bytes());
        digest.update(self.encoding_error_bound.to_bytes_be());
        digest.update((self.operations_per_drop as u64).to_le_bytes());
        for level in &self.levels {
            digest.update(level.ring_dimension().to_le_bytes());
            digest.update((level.base_bits() as u64).to_le_bytes());
            digest.update(level.modulus().to_bytes_be());
            for prime in level.to_crt().0 {
                digest.update(prime.to_le_bytes());
            }
        }
        for drop in &self.drops {
            digest.update(drop.to_bytes_be());
        }
        digest.finalize().into()
    }

    pub(crate) fn state_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        let high = self.level(level)?;
        match self.levels.get(level + 1) {
            Some(low) => {
                let base = 1u64
                    .checked_shl(low.base_bits())
                    .ok_or(RnsBggError::Parameters("state gadget base overflow"))?;
                Ok(self
                    .ring(level + 1)?
                    .gadget(1, base, low.modulus_digits())
                    .centered_extend(BigInt::from(high.modulus().as_ref().clone())))
            }
            None => Ok(self.ring(level)?.identity(1)),
        }
    }

    pub(crate) fn public_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        let parameters = self.level(level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        Ok(self.ring(level)?.gadget(1, base, parameters.modulus_digits()))
    }
}

fn public_gadget_base(base_bits: u32) -> Result<u64, RnsBggError> {
    1u64.checked_shl(base_bits).ok_or(RnsBggError::Parameters("public gadget base overflow"))
}

/// The sole descriptor shared by RNS-BGG stage checkpoints.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggArtifactBinding {
    context: [u8; 32],
    program: ExponentLutProgramId,
    parameters: [u8; 32],
    sources: Vec<mxx_ir_core::artifact::ProductionId>,
    #[serde(default)]
    lut_helpers: Vec<(crate::program::WireId, usize, usize)>,
    refresh_commitment: Option<[u8; 32]>,
}

impl RnsBggArtifactBinding {
    pub(crate) fn new(
        context: [u8; 32],
        program: ExponentLutProgramId,
        parameters: [u8; 32],
        sources: Vec<mxx_ir_core::artifact::ProductionId>,
    ) -> Result<Self, RnsBggError> {
        if sources.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RnsBggError::Binding("artifact sources must be canonical and unique"));
        }
        let mut binding = Self {
            context,
            program,
            parameters,
            sources: Vec::new(),
            lut_helpers: Vec::new(),
            refresh_commitment: None,
        };
        for source in sources {
            binding = binding.with_source(source)?;
        }
        Ok(binding)
    }

    /// Binds a refresh declaration to this stage's artifact identity.
    pub(crate) fn with_refresh_declaration(mut self, declaration: &RefreshDeclaration) -> Self {
        self.refresh_commitment = Some(*declaration.commitment());
        self
    }

    /// Attach the producing checkpoint once execution assigns its nonce.
    pub(crate) fn with_source(
        mut self,
        source: mxx_ir_core::artifact::ProductionId,
    ) -> Result<Self, RnsBggError> {
        if self.sources.binary_search(&source).is_ok() {
            return Err(RnsBggError::Binding("artifact sources must be unique"));
        }
        self.sources.push(source);
        self.sources.sort();
        Ok(self)
    }

    pub(crate) fn with_lut_helper(
        mut self,
        origin: crate::program::WireId,
        level: usize,
        count: usize,
    ) -> Result<Self, RnsBggError> {
        if count == 0 || self.lut_helpers.iter().any(|(wire, _, _)| *wire == origin) {
            return Err(RnsBggError::Binding("LUT helper metadata is not unique"));
        }
        self.lut_helpers.push((origin, level, count));
        self.lut_helpers.sort_by_key(|(wire, _, _)| *wire);
        Ok(self)
    }

    pub(crate) fn context(&self) -> &[u8; 32] {
        &self.context
    }

    pub(crate) fn program(&self) -> ExponentLutProgramId {
        self.program
    }

    pub(crate) fn parameters(&self) -> &[u8; 32] {
        &self.parameters
    }

    pub(crate) fn sources(&self) -> &[mxx_ir_core::artifact::ProductionId] {
        &self.sources
    }

    pub(crate) fn refresh_commitment(&self) -> Option<&[u8; 32]> {
        self.refresh_commitment.as_ref()
    }

    pub(crate) fn lut_helpers(&self) -> &[(crate::program::WireId, usize, usize)] {
        &self.lut_helpers
    }
}

/// Small, one-shot refresh invocation shared by the three stage lowerers.
pub struct RnsBggRefresh<'a, E: ExponentLUTProgramLowering> {
    lowerer: &'a mut E,
    inputs: BTreeMap<usize, E::Wire>,
    declaration: &'a RefreshDeclaration,
}

impl<'a, E: ExponentLUTProgramLowering> RnsBggRefresh<'a, E> {
    pub fn new(
        lowerer: &'a mut E,
        inputs: BTreeMap<usize, E::Wire>,
        declaration: &'a RefreshDeclaration,
    ) -> Result<Self, RnsBggError> {
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        if inputs.len() != declaration.slots().len() ||
            inputs.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        // RefreshDeclaration owns and validates slots, digits, labels, CRT
        // divisibility, and the exact structural PRF output mapping.  Keep
        // only that immutable declaration here; the invocation carries no
        // second copy of any refresh metadata.
        Ok(Self { lowerer, inputs, declaration })
    }

    pub(crate) fn lowerer(&mut self) -> &mut E {
        self.lowerer
    }

    pub(crate) fn inputs(&self) -> &BTreeMap<usize, E::Wire> {
        &self.inputs
    }

    pub(crate) fn slots(&self) -> &[RefreshSlot] {
        self.declaration.slots()
    }

    pub(crate) fn digits(&self) -> &RefreshDigitPlan {
        self.declaration.digits()
    }

    pub(crate) fn labels(&self) -> &RefreshPrfLabelIndex {
        self.declaration.labels()
    }

    pub(crate) fn target_level(&self) -> usize {
        self.declaration.target_level()
    }

    pub(crate) fn declaration(&self) -> &RefreshDeclaration {
        self.declaration
    }
}

fn map_row(row: RnsBggRow, mut operation: impl FnMut(Mat) -> Mat) -> RnsBggRow {
    (operation(row.0), operation(row.1))
}

fn map_matrix(matrix: RnsBggMatrix, mut operation: impl FnMut(Mat) -> Mat) -> RnsBggMatrix {
    ((operation(matrix.0.0), operation(matrix.0.1)), (operation(matrix.1.0), operation(matrix.1.1)))
}

/// Convert a matrix between concrete CRT rings without constructing an
/// invalid modulus-reduction node.  Reduction is only valid when the source
/// ring is larger; extending is used for a larger destination and equal rings
/// are preserved as-is.
pub(crate) fn checked_modulus_convert(
    value: Mat,
    destination: impl Into<BigInt>,
) -> Result<Mat, RnsBggError> {
    let destination = destination.into();
    if destination <= BigInt::zero() {
        return Err(RnsBggError::Parameters("destination modulus must be positive"));
    }
    let source = value
        .matrix_type()
        .modulus
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| RnsBggError::Parameters("source modulus must be concrete"))?;
    if source <= BigInt::zero() {
        return Err(RnsBggError::Parameters("source modulus must be positive"));
    }
    match source.cmp(&destination) {
        std::cmp::Ordering::Greater => {
            if &source % &destination != BigInt::zero() {
                return Err(RnsBggError::Parameters("destination modulus must divide source"));
            }
            Ok(value.reduce_modulus(destination))
        }
        std::cmp::Ordering::Equal => Ok(value),
        std::cmp::Ordering::Less => Ok(value.centered_extend(destination)),
    }
}

pub(crate) fn checked_modulus_convert_row(
    row: RnsBggRow,
    destination: impl Into<BigInt> + Clone,
) -> Result<RnsBggRow, RnsBggError> {
    let destination = destination.into();
    Ok((
        checked_modulus_convert(row.0, destination.clone())?,
        checked_modulus_convert(row.1, destination)?,
    ))
}

pub(crate) fn checked_modulus_convert_matrix(
    matrix: RnsBggMatrix,
    destination: impl Into<BigInt> + Clone,
) -> Result<RnsBggMatrix, RnsBggError> {
    let destination = destination.into();
    Ok((
        (
            checked_modulus_convert(matrix.0.0, destination.clone())?,
            checked_modulus_convert(matrix.0.1, destination.clone())?,
        ),
        (
            checked_modulus_convert(matrix.1.0, destination.clone())?,
            checked_modulus_convert(matrix.1.1, destination)?,
        ),
    ))
}

pub(crate) fn add_public_matrices(lhs: RnsBggMatrix, rhs: RnsBggMatrix) -> RnsBggMatrix {
    ((lhs.0.0 + rhs.0.0, lhs.0.1 + rhs.0.1), (lhs.1.0 + rhs.1.0, lhs.1.1 + rhs.1.1))
}

pub(crate) fn add_encoding_rows(lhs: RnsBggRow, rhs: RnsBggRow) -> RnsBggRow {
    (lhs.0 + rhs.0, lhs.1 + rhs.1)
}

pub(crate) fn multiply_public_action(lhs: RnsBggMatrix, rhs: RnsBggDecomposition) -> RnsBggMatrix {
    let ((a, b), (c, d)) = lhs;
    let ((u, v), (w, x)) = rhs;
    (
        (
            u.clone().mul_small_rhs(a.clone()) + w.clone().mul_small_rhs(b.clone()),
            v.clone().mul_small_rhs(a.clone()) + x.clone().mul_small_rhs(b.clone()),
        ),
        (
            u.mul_small_rhs(c.clone()) + w.mul_small_rhs(d.clone()),
            v.mul_small_rhs(c) + x.mul_small_rhs(d),
        ),
    )
}

pub(crate) fn multiply_encoding_action(lhs: RnsBggRow, rhs: RnsBggDecomposition) -> RnsBggRow {
    let (a, b) = lhs;
    let ((u, v), (w, x)) = rhs;
    (
        u.mul_small_rhs(a.clone()) + w.mul_small_rhs(b.clone()),
        v.mul_small_rhs(a) + x.mul_small_rhs(b),
    )
}

pub(crate) fn mod_down_public(value: RnsBggMatrix, modulus: BigInt) -> RnsBggMatrix {
    map_matrix(value, |entry| entry.block_mod_switch(modulus.clone(), 1))
}

pub(crate) fn mod_down_encoding(value: RnsBggRow, modulus: BigInt) -> RnsBggRow {
    map_row(value, |entry| entry.block_mod_switch(modulus.clone(), 1))
}

pub(crate) fn recompose_public(
    values: Family<RnsBggMatrix>,
    coefficients: Family<Mat>,
) -> Result<RnsBggMatrix, RnsBggError> {
    if values.count() != coefficients.count() {
        return Err(RnsBggError::Binding("CRT value and coefficient families differ"));
    }
    let weighted = mxx_dsl::parallel(values.count().clone(), |index| {
        let coefficient = coefficients.at(index.clone());
        Ok(map_matrix(values.at(index), |entry| entry * coefficient.clone()))
    })?;
    crate::utils::balanced_sum_family(weighted, add_public_matrices)
        .map_err(|error| RnsBggError::Graph(error.to_string()))
}

pub(crate) fn recompose_encoding(
    values: Family<RnsBggRow>,
    coefficients: Family<Mat>,
) -> Result<RnsBggRow, RnsBggError> {
    if values.count() != coefficients.count() {
        return Err(RnsBggError::Binding("CRT value and coefficient families differ"));
    }
    let weighted = mxx_dsl::parallel(values.count().clone(), |index| {
        let coefficient = coefficients.at(index.clone());
        let row = values.at(index);
        Ok((coefficient.clone() * row.0, coefficient * row.1))
    })?;
    crate::utils::balanced_sum_family(weighted, add_encoding_rows)
        .map_err(|error| RnsBggError::Graph(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::GraphValue;
    use mxx_ir_core::IntExpr;
    use num_traits::One;

    #[test]
    fn parameter_drops_are_exact_and_terminal_is_one() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low], 1, 2).unwrap();
        assert_eq!(
            parameters.scale(0).unwrap(),
            &(high.modulus().as_ref() / parameters.level(1).unwrap().modulus().as_ref())
        );
        assert_eq!(parameters.scale(1).unwrap(), &BigUint::one());
    }

    #[test]
    fn equal_levels_have_unit_drop() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone(), level], 1, 1).unwrap();
        assert_eq!(parameters.scale(0).unwrap(), &BigUint::one());
        assert_eq!(parameters.scale(1).unwrap(), &BigUint::one());
    }

    #[test]
    fn public_gadget_rejects_shift_overflow() {
        assert!(matches!(
            public_gadget_base(u64::BITS),
            Err(RnsBggError::Parameters("public gadget base overflow"))
        ));
    }

    #[test]
    fn checked_modulus_conversion_handles_greater_equal_and_less() {
        let source_ring = Ring::new(BigInt::from(105u8), 4);
        let source = source_ring.input("checked-modulus-source", (1, 1));
        let reduced = checked_modulus_convert(source.clone(), BigInt::from(15u8)).unwrap();
        let equal = checked_modulus_convert(source.clone(), BigInt::from(105u8)).unwrap();
        let less_source = Ring::new(BigInt::from(15u8), 4).input("checked-modulus-less", (1, 1));
        let extended = checked_modulus_convert(less_source, BigInt::from(105u8)).unwrap();

        assert_eq!(reduced.matrix_type().modulus, IntExpr::constant(BigInt::from(15u8)));
        assert_eq!(equal.flatten(), source.flatten());
        assert_eq!(extended.matrix_type().modulus, IntExpr::constant(BigInt::from(105u8)));

        let graph = mxx_dsl::DslContext::new("checked-modulus-conversion")
            .artifact_output("reduced", reduced)
            .unwrap()
            .artifact_output("equal", equal)
            .unwrap()
            .artifact_output("extended", extended)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&mxx_ir_core::ParamEnv::default()).unwrap();
    }

    #[test]
    fn lut_helper_artifact_name_preserves_root_stability() {
        let scope = FamilyScope::new(Vec::new(), Vec::new()).unwrap();
        assert_eq!(lut_helper_artifact_name(WireId(7), &scope), "lut-data/WireId(7)");
    }

    #[test]
    fn lut_helper_artifact_name_distinguishes_nonzero_and_nested_scopes() {
        let origin = WireId(11);
        let root = FamilyScope::new(Vec::new(), Vec::new()).unwrap();
        let zero = FamilyScope::new(
            vec![
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(0),
                    IntExpr::constant(2),
                    IntExpr::constant(4),
                )
                .unwrap(),
            ],
            vec![IntExpr::constant(0)],
        )
        .unwrap();
        let nonzero = FamilyScope::new(
            vec![
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(1),
                    IntExpr::constant(2),
                    IntExpr::constant(4),
                )
                .unwrap(),
            ],
            vec![IntExpr::constant(1)],
        )
        .unwrap();
        let nested_a = FamilyScope::new(
            vec![
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(2),
                    IntExpr::constant(3),
                    IntExpr::constant(8),
                )
                .unwrap(),
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(1),
                    IntExpr::constant(2),
                    IntExpr::constant(4),
                )
                .unwrap(),
            ],
            vec![IntExpr::constant(2), IntExpr::constant(1)],
        )
        .unwrap();
        let nested_b = FamilyScope::new(
            vec![
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(3),
                    IntExpr::constant(3),
                    IntExpr::constant(8),
                )
                .unwrap(),
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(1),
                    IntExpr::constant(2),
                    IntExpr::constant(4),
                )
                .unwrap(),
            ],
            vec![IntExpr::constant(3), IntExpr::constant(1)],
        )
        .unwrap();

        let root_name = lut_helper_artifact_name(origin, &root);
        let zero_name = lut_helper_artifact_name(origin, &zero);
        let nonzero_name = lut_helper_artifact_name(origin, &nonzero);
        let nested_a_name = lut_helper_artifact_name(origin, &nested_a);
        let nested_b_name = lut_helper_artifact_name(origin, &nested_b);

        assert_ne!(root_name, zero_name);
        assert_ne!(zero_name, nonzero_name);
        assert_ne!(nested_a_name, nested_b_name);
        assert!(nested_a_name.contains("start=Const(2)"));
        assert!(nested_a_name.contains("count=Const(3)"));
        assert!(nested_a_name.contains("start=Const(1)"));
        assert!(nested_a_name.contains("count=Const(2)"));
    }

    #[test]
    fn artifact_binding_rejects_duplicate_or_empty_lut_helper_metadata() {
        let mut binding = RnsBggArtifactBinding::new(
            [0; 32],
            ExponentLutProgramId::from_digest([1; 32]),
            [2; 32],
            Vec::new(),
        )
        .unwrap();
        assert!(matches!(
            binding.clone().with_lut_helper(crate::program::WireId(3), 0, 0),
            Err(RnsBggError::Binding("LUT helper metadata is not unique"))
        ));
        binding = binding.with_lut_helper(crate::program::WireId(3), 0, 1).unwrap();
        assert!(matches!(
            binding.with_lut_helper(crate::program::WireId(3), 0, 1),
            Err(RnsBggError::Binding("LUT helper metadata is not unique"))
        ));
    }
}
